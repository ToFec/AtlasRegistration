"""
Created on Oct 8, 2024

@author: fechter
"""
import unittest
import SimpleITK as sitk
import atlas_utils as atlasUtils


class Test(unittest.TestCase):
    def getDeffield(self, defField, spacing, direction):
        defField = defField[0, ...].detach().clone()
        defField = defField.permute([3, 2, 1, 0])

        defField[..., 0] = defField[..., 0] * ((defField.shape[2] - 1) / 2.0)
        defField[..., 1] = defField[..., 1] * ((defField.shape[1] - 1) / 2.0)
        defField[..., 2] = defField[..., 2] * ((defField.shape[0] - 1) / 2.0)

        defField[..., 0] = defField[..., 0] * spacing[0] * direction[0]
        defField[..., 1] = defField[..., 1] * spacing[1] * direction[4]
        defField[..., 2] = defField[..., 2] * spacing[2] * direction[8]
        defField = defField.permute([3, 2, 1, 0])
        return defField

    def testJacobian5(self):
        deformationFieldName0 = "./resources/JacobianTest/radialVF.nrrd"

        regSitk = sitk.ReadImage(deformationFieldName0)
        defFieldAtlas = atlasUtils.loadDefField(deformationFieldName0)

        determinantSITK = sitk.DisplacementFieldJacobianDeterminant(regSitk)

        meshStepLength0 = 2 / (defFieldAtlas.shape[2] - 1)
        meshStepLength1 = 2 / (defFieldAtlas.shape[3] - 1)
        meshStepLength2 = 2 / (defFieldAtlas.shape[4] - 1)

        atlasJacobi = atlasUtils.jacobianDeterminant(defFieldAtlas, (meshStepLength0, meshStepLength1, meshStepLength2))

        atlasUtils.saveImageTensor(
            atlasJacobi,
            "./resources/JacobianTest/radialVFJacobian.mha",
            regSitk.GetOrigin(),
            regSitk.GetSpacing(),
            regSitk.GetDirection(),
        )
        sitk.WriteImage(determinantSITK, "./resources/JacobianTest/sitkJacobianRadialVF.mha")

    def _testJacobian4(self):
        deformationFieldName0 = "./resources/JacobianTest/img1DefField.mha"
        regSitk = sitk.ReadImage(deformationFieldName0)
        defFieldAtlas = atlasUtils.loadDefField(deformationFieldName0)
        deformationFieldName1 = "./resources/JacobianTest/img1DefField1.mha"
        defFieldAtlas1 = atlasUtils.loadDefField(deformationFieldName1)
        newDefField = defFieldAtlas + defFieldAtlas1
        atlasUtils.saveDefField(
            "./resources/JacobianTest/img1DefFieldNew.mha",
            newDefField,
            regSitk.GetOrigin(),
            regSitk.GetSpacing(),
            regSitk.GetDirection(),
        )

    def _testJacobian3(self):
        deformationFieldName0 = "./resources/JacobianTest/img1DefField.mha"

        regSitk = sitk.ReadImage(deformationFieldName0)
        defFieldAtlas = atlasUtils.loadDefField(deformationFieldName0)

        scaledDefFieldAtlas = self.getDeffield(defFieldAtlas, regSitk.GetSpacing(), regSitk.GetDirection())
        regSitkA = sitk.GetArrayFromImage(regSitk)
        determinantSITK = sitk.DisplacementFieldJacobianDeterminant(regSitk)

        meshStepLength0 = 2 / (defFieldAtlas.shape[2] - 1)
        meshStepLength1 = 2 / (defFieldAtlas.shape[3] - 1)
        meshStepLength2 = 2 / (defFieldAtlas.shape[4] - 1)

        atlasJacobiScaled = atlasUtils.jacobianDeterminant(scaledDefFieldAtlas[None, ...], regSitk.GetSpacing())
        atlasJacobi = atlasUtils.jacobianDeterminant(defFieldAtlas, (meshStepLength0, meshStepLength1, meshStepLength2))

        atlasUtils.saveImageTensor(
            atlasJacobi,
            "./resources/JacobianTest/atlasScaledJacobian.mha",
            regSitk.GetOrigin(),
            regSitk.GetSpacing(),
            regSitk.GetDirection(),
        )
        atlasUtils.saveDefField(
            "./resources/JacobianTest/atlasDeffieldSaved.mha",
            defFieldAtlas,
            regSitk.GetOrigin(),
            regSitk.GetSpacing(),
            regSitk.GetDirection(),
        )

        self.assertAlmostEqual(atlasJacobiScaled[0, 0, 24, 16, 17].item(), atlasJacobi[0, 0, 24, 16, 17].item(), 5)

    def _testJacobian2(self):
        deformationFieldName1 = "./resources/JacobianTest/img1DefField1.mha"

        defFieldAtlas1 = atlasUtils.loadDefField(deformationFieldName1)
        regSitk = sitk.ReadImage(deformationFieldName1)
        scaledDefFieldAtlas1 = self.getDeffield(defFieldAtlas1, regSitk.GetSpacing(), regSitk.GetDirection())

        meshStepLength0 = 2 / (defFieldAtlas1.shape[2] - 1)
        meshStepLength1 = 2 / (defFieldAtlas1.shape[3] - 1)
        meshStepLength2 = 2 / (defFieldAtlas1.shape[4] - 1)

        atlasJacobiScaled = atlasUtils.jacobianDeterminant(scaledDefFieldAtlas1[None, ...], regSitk.GetSpacing())
        atlasJacobi = atlasUtils.jacobianDeterminant(
            defFieldAtlas1, (meshStepLength0, meshStepLength1, meshStepLength2)
        )

        atlasUtils.saveImageTensor(
            atlasJacobi,
            "./resources/JacobianTest/atlasScaledJacobian1.mha",
            regSitk.GetOrigin(),
            regSitk.GetSpacing(),
            regSitk.GetDirection(),
        )
        self.assertAlmostEqual(atlasJacobiScaled[0, 0, 24, 16, 17].item(), atlasJacobi[0, 0, 24, 16, 17].item(), 5)

    def _testJacobian(self):
        deformationFieldName = "./resources/JacobianTest/img1DefField.mha"
        regSitk = sitk.ReadImage(deformationFieldName)
        defFieldAtlas = atlasUtils.loadDefField(deformationFieldName)

        scaledDefFieldAtlas = self.getDeffield(defFieldAtlas, regSitk.GetSpacing(), regSitk.GetDirection())
        atlasJacobi = atlasUtils.jacobianDeterminant(scaledDefFieldAtlas[None, ...], regSitk.GetSpacing())
        atlasUtils.saveImageTensor(
            atlasJacobi,
            "./resources/JacobianTest/atlasScaledJacobian.mha",
            regSitk.GetOrigin(),
            regSitk.GetSpacing(),
            regSitk.GetDirection(),
        )

        meshStepLength0 = 2 / (defFieldAtlas.shape[2] - 1)
        meshStepLength1 = 2 / (defFieldAtlas.shape[3] - 1)
        meshStepLength2 = 2 / (defFieldAtlas.shape[4] - 1)

        determinantSITK = sitk.DisplacementFieldJacobianDeterminant(regSitk)

        sitk.WriteImage(determinantSITK, "./resources/JacobianTest/sitkJacobian.mha")

        direction = regSitk.GetDirection()
        defFieldAtlas[:, 0, ...] = defFieldAtlas[:, 0, ...] * direction[0]
        defFieldAtlas[:, 1, ...] = defFieldAtlas[:, 1, ...] * direction[1]
        defFieldAtlas[:, 2, ...] = defFieldAtlas[:, 2, ...] * direction[2]
        atlasJacobi = atlasUtils.jacobianDeterminant(defFieldAtlas, (meshStepLength0, meshStepLength1, meshStepLength2))
        atlasUtils.saveImageTensor(
            atlasJacobi,
            "./resources/JacobianTest/atlasJacobian.mha",
            regSitk.GetOrigin(),
            regSitk.GetSpacing(),
            regSitk.GetDirection(),
        )

        atlasUtils.saveDefField(
            "./resources/JacobianTest/atlasDeffieldSaved.mha",
            defFieldAtlas,
            regSitk.GetOrigin(),
            regSitk.GetSpacing(),
            regSitk.GetDirection(),
        )


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testJacobian']
    unittest.main()
